CARND_IP = '54.218.123.124'
# INSTANCE_ID = 'i-516ff7c4'

namespace :carnd do
  task :ssh do
    sh "ssh carnd@#{CARND_IP}"
  end

  task :pull do
    sh "ssh -t carnd@#{CARND_IP} 'cd ~/carnd-behavioral-cloning && git pull'"
  end

  task :train, [:network] => :pull do |t, args|
    args.with_defaults(network: '~/carnd-keras-lab/networks/keras/german_traffic_sign_cnn.py')
    sh "ssh -t carnd@#{CARND_IP} 'python3 #{args[:network]}'"
  end

  task :drive do |t, args|
    sh 'python drive.py model.json'
  end

  task :scp, [:src, :dest] do |t, args|
    args.with_defaults(dest: '~')
    host = "carnd@#{CARND_IP}"
    puts "downloading #{args[:src]} to #{host}:#{args[:dest]}"
    sh "scp -rp #{args[:src]} #{host}:#{args[:dest]}"
  end

  task :rsync, [:src, :dest] do |t, args|
    args.with_defaults(dest: '~')
    host = "carnd@#{CARND_IP}"
    puts "uploading #{args[:src]} to #{host}:#{args[:dest]}"
    # %w(zimpy networks).each do |file_or_dir|
    #   sh "rsync -avz --exclude '*.zip' --exclude '*.pickle' --exclude '*.p' #{file_or_dir} #{host}:#{args[:dest]}"
    # end
    unless args[:src].nil?
      sh "rsync -avz #{args[:src]} #{host}:#{args[:dest]}"
    end
  end

  task :get_model, [] do
    sh "ssh -t carnd@#{CARND_IP} 'zip -r ~/trained_model.zip carnd-behavioral-cloning/model.json carnd-behavioral-cloning/model.h5'"
    sh "scp -rp carnd@#{CARND_IP}:~/trained_model.zip ."
    sh 'unzip -u trained_model.zip -d ..'
  end

  task :down, [:src, :dest] do |t, args|
    args.with_defaults(dest: '~')
    host = "carnd@#{CARND_IP}"
    puts "downloading #{host}:#{args[:src]} to #{args[:dest]}"
    sh "scp -rp #{host}:#{args[:src]} #{args[:dest]}"
  end

  # task :start, [:instance_id] do |t, args|
  #   args.with_defaults(instance_id: INSTANCE_ID)
  #   instance_id = args[:instance_id]
  #   sh "aws ec2 start-instances --instance-ids \"#{instance_id}\""
  # end

  # task :stop, [:instance_id] do |t, args|
  #   args.with_defaults(instance_id: INSTANCE_ID)
  #   instance_id = args[:instance_id]
  #   sh "aws ec2 stop-instances --instance-ids \"#{instance_id}\""
  # end
end